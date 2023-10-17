import numpy as np
import sklearn.metrics.pairwise as smp


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers

    def score_gradients(self, local_grads, selectec_peers):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, grad_len))

        grads = np.zeros((m, grad_len))
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory[selectec_peers] += grads
        wv = foolsgold(self.memory)  # Use FG
        self.wv_history.append(wv)
        return wv[selectec_peers]


# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    print("---------running fools gold--------------")
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv
