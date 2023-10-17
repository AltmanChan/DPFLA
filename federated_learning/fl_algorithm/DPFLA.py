import copy
import time
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import sklearn.metrics.pairwise as smp
from sklearn.preprocessing import StandardScaler


class DPFLA:
    def __init__(self):
        pass

    def score(self, global_model, local_models, clients_types, selected_clients, p, w):

        n = len(selected_clients)
        P = generate_orthogonal_matrix(n=p, reuse=True)
        W = generate_orthogonal_matrix(n=n * n, reuse=True)
        Ws = [W[:, e * n: e * n + n][0, :].reshape(-1, 1) for e in range(n)]

        param_diff = []
        param_diff_mask = []

        m_len = len(local_models)

        detect_res_list = []
        start_model_layer_param_list = []

        for idx in range(10):
            start_model_layer_param_list.append(list(global_model.state_dict()['fc2.weight'][idx].cpu()))
            # 计算每个本地模型的权重与全局模型最后一层权重之间的梯度差
            for i in range(m_len):
                end_model_layer_param = list(local_models[i].state_dict()['fc2.weight'][idx].cpu())
                gradient = calculate_parameter_gradients(start_model_layer_param_list[idx], end_model_layer_param)
                gradient = gradient.flatten()
                X_mask = Ws[i] @ gradient.reshape(1, -1) @ P
                param_diff_mask.append(X_mask)

            Z_mask = sum(param_diff_mask)
            U_mask, sigma, VT_mask = svd(Z_mask)

            G = Ws[0]
            for idx, val in enumerate(selected_clients):
                if idx == 0:
                    continue
                G = np.concatenate((G, Ws[idx]), axis=1)

            U = np.linalg.inv(G) @ U_mask
            U = U[:, :2]
            res = U * sigma[:2]
            detect_res_list.append(res)

        coefficient_list, score_list = batch_detect_outliers_kmeans(detect_res_list)

        max_sc = max(coefficient_list)
        max_sc_idx = coefficient_list.index(max_sc)
        scores = score_list[max_sc_idx] if max_sc >= 0.70 else np.ones(n, dtype=int)

        logger.debug("-------------------------------------")
        logger.debug("Max Silhouette Coefficient: " + str(max_sc))
        logger.debug("Detect Class: " + str(max_sc_idx))
        logger.debug("Defense result:")
        for i, pt in enumerate(clients_types):
            logger.info(str(pt) + ' scored ' + str(scores[i]))

        # 返回得分列表
        return scores


def generate_orthogonal_matrix(n, reuse=False, block_size=None):
    orthogonal_matrix_cache_dir = 'orthogonal_matrices'
    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    file_name = str(n)
    if block_size is not None:
        file_name += '_blc%s' % block_size

    if reuse and file_name in existing:
        with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        if block_size is not None:
            qs = [block_size] * int(n / block_size)
            if n % block_size != 0:
                qs[-1] += (n - np.sum(qs))
            q = np.zeros([n, n])
            for i in range(len(qs)):
                sub_n = qs[i]
                tmp = generate_orthogonal_matrix(sub_n, reuse=False, block_size=sub_n)
                index = int(np.sum(qs[:i]))
                q[index:index + sub_n, index:index + sub_n] += tmp
        else:
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        if reuse:
            with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(q, f, protocol=4)
        return q


def calculate_parameter_gradients(params_1, params_2):
    return np.array([x for x in np.subtract(params_1, params_2)])


def detect_outliers_kmeans(data, n_clusters=2):
    # 初始化K-means模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # 训练模型
    kmeans.fit(data)
    # 预测聚类标签
    labels = kmeans.predict(data)
    # 计算轮廓系数
    coefficient = silhouette_score(data, labels)
    logger.debug("Silhouette Coefficient：{}", coefficient)
    # calinski_harabasz = calinski_harabasz_score(data, labels)
    # logger.debug("Calinski Harabasz：{}", calinski_harabasz)
    if coefficient < 0.61:
        return np.ones(len(data), dtype=int)

    scores = labels
    if sum(labels) < len(data) / 2:
        scores = 1 - labels
    else:
        scores = labels

    return scores


def batch_detect_outliers_kmeans(list, n_clusters=2):
    # 初始化K-means模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    coefficient_list = []
    score_list = []

    for data in list:
        # 训练模型
        kmeans.fit(data)
        # 预测聚类标签
        labels = kmeans.predict(data)
        # 计算轮廓系数
        coefficient = silhouette_score(data, labels)
        coefficient_list.append(coefficient)
        logger.debug("Silhouette Coefficient：{}", coefficient)

        scores = labels
        if sum(labels) < len(data) / 2:
            scores = 1 - labels
        else:
            scores = labels

        score_list.append(scores)
    return coefficient_list, score_list


def svd(x):
    m, n = x.shape
    if m >= n:
        return np.linalg.svd(x)
    else:
        u, s, v = np.linalg.svd(x.T)
        return v.T, s, u.T


def draw(data, clients_types, scores):
    SAVE_NAME = str(time.time()) + '.jpg'

    fig = plt.figure(figsize=(20, 6))
    fig1 = plt.subplot(121)
    for i, pt in enumerate(clients_types):
        if pt == 'Good update':
            plt.scatter(data[i, 0], data[i, 1], facecolors='none', edgecolors='black', marker='o', s=800,
                        label="Good update")
        else:
            plt.scatter(data[i, 0], data[i, 1], facecolors='black', edgecolors='black', marker='o', s=800,
                        label="Bad update")

    fig2 = plt.subplot(122)
    for i, pt in enumerate(clients_types):
        if scores[i] == 1:
            plt.scatter(data[i, 0], data[i, 1], color="orange", s=800, label="Good update")
        else:
            plt.scatter(data[i, 0], data[i, 1], color="blue", marker="x", linewidth=3, s=800, label="Bad update")

    plt.grid(False)
    # plt.show()
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1, dpi=400)
