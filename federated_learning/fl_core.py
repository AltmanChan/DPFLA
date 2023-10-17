import time
import copy
import numpy as np
import random
import torch
import os
from loguru import logger

from torch.utils.data import DataLoader
from sklearn.metrics import *
import gc
from tqdm import tqdm_notebook

from federated_learning.client import Client
from federated_learning.datasets import CustomDataset
from federated_learning.fl_algorithm import FoolsGold, MyFunction, average_weights, simple_median, trimmed_mean, Krum, \
    FedSVD, LFD, FLDefender, DPFLA
from federated_learning.models import setup_model
from federated_learning.utils import distribute_dataset, contains_class


class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_clients, frac_clients,
                 seed, test_batch_size, criterion, global_rounds, local_epochs, local_bs, local_lr,
                 local_momentum, labels_dict, device, attackers_ratio=0,
                 class_per_client=2, samples_per_class=250, rate_unbalance=1, alpha=1, source_class=None):

        FL._history = np.zeros(num_clients)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_clients = num_clients
        self.clients_pseudonyms = ['Client ' + str(i + 1) for i in range(self.num_clients)]
        self.frac_clients = frac_clients
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.class_per_client = class_per_client
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.clients = []
        self.trainset, self.testset = None, None

        self.score_history = np.zeros([self.num_clients], dtype=float)

        # Fix the random state of the environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        # Loading of data
        self.trainset, self.testset, user_groups_train, tokenizer = distribute_dataset(self.dataset_name,
                                                                                       self.num_clients,
                                                                                       self.num_classes,
                                                                                       self.dd_type,
                                                                                       self.class_per_client,
                                                                                       self.samples_per_class,
                                                                                       self.alpha)

        self.test_loader = DataLoader(self.testset, batch_size=self.test_batch_size,
                                      shuffle=False, num_workers=1)

        # Creating model
        self.global_model = setup_model(model_architecture=self.model_name, num_classes=self.num_classes,
                                        tokenizer=tokenizer, embedding_dim=self.embedding_dim)
        self.global_model = self.global_model.to(self.device)

        # Dividing the training set among clients
        self.local_data = []
        self.have_source_class = []
        self.labels = []
        logger.info('--> Distributing training data among clients')
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            client_data = CustomDataset(self.trainset, indices=indices)
            self.local_data.append(client_data)
            if self.source_class in user_groups_train[p]['labels']:
                self.have_source_class.append(p)
        logger.info('--> Training data have been distributed among clients')

        # Creating clients instances
        logger.info('--> Creating peets instances')
        m_ = 0
        if self.attackers_ratio > 0:
            # pick m random participants from the workers list
            # k_src = len(self.have_source_class)
            # print('# of clients who have source class examples:', k_src)
            m_ = int(self.attackers_ratio * self.num_clients)
            self.num_attackers = copy.deepcopy(m_)

        clients = list(np.arange(self.num_clients))
        random.shuffle(clients)
        for i in clients:
            if m_ > 0 and contains_class(self.local_data[i], self.source_class):
                self.clients.append(Client(i, self.clients_pseudonyms[i],
                                           self.local_data[i], self.labels[i],
                                           self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr,
                                           self.local_momentum, client_type='attacker'))
                m_ -= 1
            else:
                self.clients.append(Client(i, self.clients_pseudonyms[i],
                                           self.local_data[i], self.labels[i],
                                           self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr,
                                           self.local_momentum))

        del self.local_data

    # ======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader, dataset_name=None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            if dataset_name == 'IMDB':
                test_loss.append(self.criterion(output, target.view(-1, 1)).item())  # sum up batch loss
                pred = output > 0.5  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                test_loss.append(self.criterion(output, target).item())  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            n += target.shape[0]
        test_loss = np.mean(test_loss)
        logger.debug('Average test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, n,
                                                                                        100 * correct / n))
        return 100.0 * (float(correct) / n), test_loss

    # ======================================= End of testning function =============================================================#
    # Test label prediction function
    def test_label_predictions(self, model, device, test_loader, dataset_name=None):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if dataset_name == 'IMDB':
                    prediction = output > 0.5
                else:
                    prediction = output.argmax(dim=1, keepdim=True)

                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]

    # choose random set of clients
    def choose_clients(self):
        # pick m random clients from the available list of clients
        m = max(int(self.frac_clients * self.num_clients), 1)
        selected_clients = np.random.choice(range(self.num_clients), m, replace=False)

        # print('\nSelected Clients\n')
        # for i, p in enumerate(selected_clients):
        #     print(i+1, ': ', self.clients[p].client_pseudonym, ' is ', self.clients[p].client_type)
        return selected_clients

    def test_backdoor(self, model, device, test_loader, backdoor_pattern, source_class, target_class):
        model.eval()
        correct = 0
        n = 0
        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            keep_idxs = (target == source_class)
            bk_data = copy.deepcopy(data[keep_idxs])
            bk_target = copy.deepcopy(target[keep_idxs])
            bk_data[:, :, -x_offset:, -y_offset:] = backdoor_pattern
            bk_target[:] = target_class
            output = model(bk_data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(bk_target.view_as(pred)).sum().item()
            n += bk_target.shape[0]
        return np.round(100.0 * (float(correct) / n), 2)

    def run_experiment(self, attack_type='no_attack', malicious_behavior_rate=0,
                       source_class=None, target_class=None, rule='fedavg', resume=False, model_name=None,
                       untarget=False):
        simulation_model = copy.deepcopy(self.global_model)

        logger.info("===>Simulation started...")

        lfd = LFD(self.num_classes)
        fg = FoolsGold(self.num_clients)
        my_function = MyFunction()
        fed_svd = FedSVD()
        fl_dfndr = FLDefender(self.num_clients)
        dpfla = DPFLA()
        # copy weights
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        noise_scalar = 1.0
        # best_accuracy = 0.0
        mapping = {'honest': 'Good update', 'attacker': 'Bad update'}

        # start training
        start_round = 0
        if resume:
            logger.info('Loading last saved checkpoint..')
            checkpoint = torch.load(
                './checkpoints/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
                    self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7')
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']

            logger.info('>>checkpoint loaded!')
        logger.info("====>Global model training started...")
        for epoch in tqdm_notebook(range(start_round, self.global_rounds)):
            gc.collect()
            torch.cuda.empty_cache()

            # if epoch % 20 == 0:
            #     clear_output()
            logger.debug(f'| Global training round : {epoch + 1}/{self.global_rounds} |')
            selected_clients = self.choose_clients()
            local_weights, local_grads, local_models, local_losses, performed_attacks = [], [], [], [], []
            clients_types = []
            i = 1
            attacks = 0
            Client._performed_attacks = 0
            for client in selected_clients:
                clients_types.append(mapping[self.clients[client].client_type])
                # print(i)
                # print('\n{}: {} Starts training in global round:{} |'.format(i, (self.clients_pseudonyms[client]), (epoch + 1)))
                client_update, client_grad, client_local_model, client_loss, attacked, t = self.clients[
                    client].participant_update(
                    epoch,
                    copy.deepcopy(simulation_model),
                    untarget=untarget,
                    attack_type=attack_type, malicious_behavior_rate=malicious_behavior_rate,
                    source_class=source_class, target_class=target_class, dataset_name=self.dataset_name)
                local_weights.append(client_update)
                local_grads.append(client_grad)
                local_losses.append(client_loss)
                local_models.append(client_local_model)
                attacks += attacked
                # print('{} ends training in global round:{} |\n'.format((self.clients_pseudonyms[client]), (epoch + 1)))
                i += 1
            # loss_avg = sum(local_losses) / len(local_losses)
            # print('Average of clients\' local losses: {:.6f}'.format(loss_avg))
            # aggregated global weights
            scores = np.zeros(len(local_weights))
            # Expected malicious clients
            f = int(self.num_clients * self.attackers_ratio)
            if rule == 'median':
                cur_time = time.time()
                global_weights = simple_median(local_weights)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'tmean':
                cur_time = time.time()
                # trim_ratio = self.attackers_ratio * self.num_clients / len(selected_clients)
                trim_ratio = self.attackers_ratio * self.num_clients / len(selected_clients)
                global_weights = trimmed_mean(local_weights, trim_ratio=trim_ratio)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'mkrum':
                cur_time = time.time()
                good_updates = Krum(local_models, f=f, multi=True)
                scores[good_updates] = 1
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'foolsgold':
                cur_time = time.time()
                scores = fg.score_gradients(local_grads, selected_clients)

                logger.debug("Defense result:")
                for i, pt in enumerate(clients_types):
                    logger.info(str(pt) + ' scored ' + str(scores[i]))

                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time + t)

            elif rule == 'fl_defender':
                cur_time = time.time()
                scores = fl_dfndr.score(copy.deepcopy(simulation_model),
                                        copy.deepcopy(local_models),
                                        peers_types=clients_types,
                                        selected_peers=selected_clients,
                                        epoch=epoch + 1,
                                        tau=(1.5 * epoch / self.global_rounds))

                trust = self.update_score_history(scores, selected_clients, epoch)
                t = time.time() - cur_time

                logger.debug("Defense result:")
                for i, pt in enumerate(clients_types):
                    logger.info(str(pt) + ' scored ' + str(trust[i]))

                global_weights = average_weights(local_weights, trust)
                cpu_runtimes.append(t)

            elif rule == 'fedavg':
                cur_time = time.time()
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'my_function':
                cur_time = time.time()
                if model_name == "CNNMNIST":
                    m = 50
                    n = 10
                elif model_name == "CNNCifar10":
                    m = 128
                    n = 10
                else:
                    raise Exception('Undefined model name!!!')

                # P = generate_orthogonal_matrix(n=m, reuse=True)
                # W = generate_orthogonal_matrix(n=n * self.num_clients, reuse=True)
                # Ws = [W[:, e * n: e * n + n][0, :].reshape(-1, 1) for e in range(self.num_clients)]

                scores = my_function.score(copy.deepcopy(simulation_model),
                                           copy.deepcopy(local_models),
                                           clients_types=clients_types,
                                           selected_clients=selected_clients, p=m, w=n)
                global_weights = average_weights(local_weights, scores)
                t = time.time() - cur_time
                logger.debug('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)

            elif rule == 'fed_svd':
                print("--------------------------")
                cur_time = time.time()
                new_global_weights = fed_svd.aggregation(copy.deepcopy(global_weights),
                                                         copy.deepcopy(local_weights),
                                                         [])
                global_weights = new_global_weights
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'DPFLA':
                cur_time = time.time()
                if model_name == "CNNMNIST":
                    m = 50
                    n = 10
                elif model_name == "CNNCifar10":
                    m = 128
                    n = 10
                else:
                    raise Exception('Undefined model name!!!')

                # P = generate_orthogonal_matrix(n=m, reuse=True)
                # W = generate_orthogonal_matrix(n=n * self.num_clients, reuse=True)
                # Ws = [W[:, e * n: e * n + n][0, :].reshape(-1, 1) for e in range(self.num_clients)]

                scores = dpfla.score(copy.deepcopy(simulation_model),
                                     copy.deepcopy(local_models),
                                     clients_types=clients_types,
                                     selected_clients=selected_clients, p=m, w=n)
                global_weights = average_weights(local_weights, scores)
                t = time.time() - cur_time
                logger.debug('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)

            else:
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                ##############################################################################################

            g_model = copy.deepcopy(simulation_model)
            simulation_model.load_state_dict(global_weights)
            if epoch >= self.global_rounds - 10:
                last10_updates.append(global_weights)

            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader,
                                                    dataset_name=self.dataset_name)

            if np.isnan(test_loss):
                simulation_model = copy.deepcopy(g_model)
                noise_scalar = noise_scalar * 0.5

            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
            performed_attacks.append(attacks)

            backdoor_asr = 0.0
            backdoor_pattern = None
            if attack_type == 'backdoor':
                if self.dataset_name == 'MNIST':
                    backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                     [2.8238, 2.8238, 2.8238],
                                                     [2.8238, 2.8238, 2.8238]])
                elif self.dataset_name == 'CIFAR10':
                    backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                      [2.5141, 2.5141, 2.5141],
                                                      [2.5141, 2.5141, 2.5141]],

                                                     [[2.5968, 2.5968, 2.5968],
                                                      [2.5968, 2.5968, 2.5968],
                                                      [2.5968, 2.5968, 2.5968]],

                                                     [[2.7537, 2.7537, 2.7537],
                                                      [2.7537, 2.7537, 2.7537],
                                                      [2.7537, 2.7537, 2.7537]]])

                backdoor_asr = self.test_backdoor(simulation_model, self.device, self.test_loader,
                                                  backdoor_pattern, source_class, target_class)
            logger.info('Backdoor ASR {}'.format(backdoor_asr))

            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(),
                'global_model': g_model,
                'local_models': copy.deepcopy(local_models),
                'last10_updates': last10_updates,
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies
            }
            # savepath = './checkpoints/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
            #     self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
            # torch.save(state, savepath)
            del local_models
            del local_weights
            del local_grads
            gc.collect()
            torch.cuda.empty_cache()
            # print("***********************************************************************************")
            # print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader,
                                                               dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())

            logger.debug('{0:10s} - {1}'.format('Class', 'Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                logger.info('{0:10s} - {1:.1f}'.format(classes[i], r[i] / np.sum(r) * 100))
                if i == source_class:
                    source_class_accuracies.append(np.round(r[i] / np.sum(r) * 100, 2))

            if epoch == self.global_rounds - 1:
                logger.info('Last 10 updates results')
                global_weights = average_weights(last10_updates,
                                                 np.ones([len(last10_updates)]))
                simulation_model.load_state_dict(global_weights)
                current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader,
                                                        dataset_name=self.dataset_name)
                global_accuracies.append(np.round(current_accuracy, 2))
                test_losses.append(np.round(test_loss, 4))
                performed_attacks.append(attacks)
                logger.info("***********************************************************************************")
                # print and show confusion matrix after each global round
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader,
                                                                   dataset_name=self.dataset_name)
                classes = list(self.labels_dict.keys())
                logger.info('{0:10s} - {1}'.format('Class', 'Accuracy'))
                asr = 0.0
                for i, r in enumerate(confusion_matrix(actuals, predictions)):
                    logger.info('{0:10s} - {1:.1f}'.format(classes[i], r[i] / np.sum(r) * 100))
                    if i == source_class:
                        source_class_accuracies.append(np.round(r[i] / np.sum(r) * 100, 2))
                        asr = np.round(r[target_class] / np.sum(r) * 100, 2)

                backdoor_asr = 0.0
                if attack_type == 'backdoor':
                    if self.dataset_name == 'MNIST':
                        backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                         [2.8238, 2.8238, 2.8238],
                                                         [2.8238, 2.8238, 2.8238]])
                    elif self.dataset_name == 'CIFAR10':
                        backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                          [2.5141, 2.5141, 2.5141],
                                                          [2.5141, 2.5141, 2.5141]],

                                                         [[2.5968, 2.5968, 2.5968],
                                                          [2.5968, 2.5968, 2.5968],
                                                          [2.5968, 2.5968, 2.5968]],

                                                         [[2.7537, 2.7537, 2.7537],
                                                          [2.7537, 2.7537, 2.7537],
                                                          [2.7537, 2.7537, 2.7537]]])

                    backdoor_asr = self.test_backdoor(simulation_model, self.device, self.test_loader,
                                                      backdoor_pattern, source_class, target_class)

        state = {
            'state_dict': simulation_model.state_dict(),
            'test_losses': test_losses,
            'global_accuracies': global_accuracies,
            'source_class_accuracies': source_class_accuracies,
            'asr': asr,
            'backdoor_asr': backdoor_asr,
            'avg_cpu_runtime': np.mean(cpu_runtimes)
        }
        # savepath = './results/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
        #     self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
        # torch.save(state, savepath)

        logger.debug('Global accuracies: {}'.format(global_accuracies))
        logger.debug('Class {} accuracies: {}'.format(source_class, source_class_accuracies))
        logger.debug("Test loss: {}".format(test_losses))
        logger.debug("Label-flipping Attack success rate: {}".format(asr))
        logger.debug('Backdoor attack succes rate: {}'.format(backdoor_asr))
        logger.debug("Average CPU aggregation runtime: {}".format(np.mean(cpu_runtimes)))

    def update_score_history(self, scores, selected_peers, epoch):
        print('-> Update score history')
        self.score_history[selected_peers] += scores
        q1 = np.quantile(self.score_history, 0.25)
        trust = self.score_history - q1
        trust = trust / trust.max()
        trust[(trust < 0)] = 0
        return trust[selected_peers]
