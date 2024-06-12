import torch
import numpy as np
# 官方的pooling调用方法
# nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))

def max_pooling(inputs, pool_size, stride):
    """
    最大池化操作
    inputs: 输入数据，形状为 (C, H, W)
    pool_size: 池化核的大小
    stride: 步长
    """
    C, H, W = inputs.shape

    H_out = (H-pool_size) // stride + 1
    W_out = (W-pool_size) // stride + 1
    
    outputs = np.zeros((C, H_out, W_out)) 
    for i in range(H_out):
        for j in range(W_out):
            input_part = inputs[:, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            outputs[:,i,j] = np.max(input_part, axis=(1,2))

    return outputs