# DPFLA

DPFLA is a defensive scheme for federated learning that can detect poisoning attacks without revealing the actual gradients of participants.

## News!

**March 01, 2024: Our Paper DPFLA Accepted by *IEEE Transactions on Services Computing*!**
Our paper "DPFLA: Defending Privacy Federated Learning Against Poisoning Attacks" is accepted by IEEE Transactions on Services Computing.

## Introduction

We propose DPFLA which adopts removable masks as the privacy protection technology and provides the server with a channel to penalty malicious participants via dimensionality reduction analysis of the final layer gradients. We evaluate DPFLA on two datasets (MNIST and CIFAR10) and under two attacks (Label-flipping attack and backdoor attack). Additionally, We compared the performance of DPFLA with five previous studies (i.e., FedAvg [1], Median [2], TMean [2], FoolsGold [3] and MKrum [4])

[1] McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017, April). Communication-efficient learning of deep networks from decentralized data. In *Artificial intelligence and statistics* (pp. 1273-1282). PMLR.

[2] Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018, July). Byzantine-robust distributed learning: Towards optimal statistical rates. In *International Conference on Machine Learning* (pp. 5650-5659). PMLR.

[3] Fung, C., Yoon, C. J., & Beschastnikh, I. (2020). The limitations of federated learning in sybil settings. In *23rd International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2020)* (pp. 301-316).

[4] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. *Advances in neural information processing systems*, *30*.

