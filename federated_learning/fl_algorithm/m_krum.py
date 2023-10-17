import torch


def Krum(updates, f, multi=False):
    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.empty([n, len(updates[0])])

    for i in range(n):
        updates_[i] = updates[i]

    k = n - f - 2
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k, largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()
    if multi:
        return idxs[:k]
    else:
        return idxs[0]
