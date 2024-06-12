import numpy as np


def batch_norm(inputs, gamma, beta, eps):
    """
    批量归一化操作。N, C, H, W样本数可以看做N*H*W，CNN中，BN通常在每个通道上独立进行
    inputs: 输入数据，形状为 (N, C, H, W)
    gamma: 缩放因子，形状为 (C,)
    beta: 偏移因子，形状为 (C,)
    eps: 防止除0的小数值
    """
    N, C, H, W = inputs.shape

    # 计算每个通道的均值&方差
    mean = np.mean(inputs, axis=(0,2,3), keepdims=True) # [1,C,1,1]
    var  = np.var(inputs, axis=(0,2,3), keepdims=True) # [1,C,1,1]

    inputs_norm = (inputs - mean) / np.sqrt(var + eps)

    outputs = gamma * inputs_norm + beta

    return outputs