import copy
from functools import reduce
import torch

from federated_learning.fl_algorithm import average_weights


def trimmed_mean(w, trim_ratio):
    print("---------running trimmed mean--------------")
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])

    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    # 对权重字典进行深拷贝，防止修改原始权重
    w_med = copy.deepcopy(w[0])
    # 对每个权重字典中的键进行操作
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:  # 如果该键对应的值是标量，则不进行修剪平均
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        # 将所有权重中该键对应的值拼接成一个二维张量，第一维是权重个数，第二维是该键对应的张量展平后的一维大小
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)  # 将张量转置，便于排序和取平均
        y_sorted = y.sort()[0]  # 对张量排序
        result = y_sorted[:, trim_num:-trim_num]  # 去掉前后trim_num个数，取平均值
        result = result.mean(dim=-1)  # 对最后一维取平均值，得到修剪平均值
        assert total_num == len(result)

        weight = torch.reshape(result, shape)  # 将修剪平均值重构为与原始张量形状相同的张量
        w_med[k] = weight  # 更新权重字典中该键对应的值
    return w_med