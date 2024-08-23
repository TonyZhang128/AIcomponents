import torch
import numpy as np

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
def conv2d(inputs, kernels, bias, stride, padding):
    """
    正向卷积操作
    inputs: 输入数据，形状为 (C, H, W)
    kernels: 卷积核，形状为 (F, C, HH, WW)，C是图片输入层数，F是图片输出层数
    bias: 偏置，形状为 (F,)
    stride: 步长
    padding: 填充
    """
    # 获取输入数据和卷积核的形状
    C, H, W = inputs.shape 
    F, C, HH, WW = kernels.shape

    # padding
    inputs_pad = np.pad(inputs, ((0,0), (padding, padding), (padding, padding)))

    # output
    H_out = (H-HH+2*padding) // stride + 1 
    W_out = (W-WW+2*padding) // stride + 1
    outputs = np.zeros((F, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            input_part = inputs_pad[:,i*stride:i*stride+HH, j*stride:j*stride+WW] # [C, HH, WW]
            outputs[:, i, j] = np.sum(input_part * kernels, axis=(1,2,3)) + bias
    
    return outputs