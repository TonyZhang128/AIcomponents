import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

def my_conv2d(inputs, out_channels, kernel_size, stride, padding):
    """
    正向卷积操作
    inputs: 输入数据，形状为 (C, H, W)
    out_channels: 输出通道数F
    kernel_size: 卷积核尺寸
    stride: 步长
    padding: 填充
    """
    # 获取输入数据和卷积核的形状
    C, H, W = inputs.shape
    F = out_channels
    HH, WW = kernel_size

    # 初始化卷积核kernel以及偏置bias
    kernels = np.random.normal(0, 1, size=(out_channels*C*HH*WW, 1))\
                                    .reshape(out_channels, C, HH, WW)
    bias = np.random.normal(0, 1, size=(out_channels, 1))

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

def my_pool(inputs, pool_size, stride):
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

class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        # 调用Pytorch官方conv2d算子
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        # 调用Pytorch官方pool2d算子
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1) 
        # 定义全连接层
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        
        # 把官方conv2d算子替换为我们实现的卷积算子
        x = my_conv2d(x, 32, 3, 1, 0)
        x = my_conv2d(x, 64, 3, 1, 0)
        # x = self.pool(x)

        # 把官方pool2d算子替换为我们实现的池化算子
        x = my_pool(x, 3, 1)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x)

        return output       

# 读入图片并转成numpy.ndarray
img = Image.open('./img/example2.jpg')
img = np.array(img)

# 调用CNN模型
model = CNNmodel() 
x = img.astype('float32')
x = x.reshape(1,1,img.shape[0], img.shape[1])

y = model(x) # 分类预测结果

def my_conv2d(inputs, out_channels, kernel_size, stride, padding):
    # 获取输入数据和卷积核的形状
    C, H, W = inputs.shape
    F = out_channels
    HH, WW = kernel_size

    # 初始化卷积核kernel以及偏置bias
    kernels = np.random.normal(0, 1, size=(out_channels*C*HH*WW, 1))\
                                    .reshape(out_channels, C, HH, WW)
    bias = np.random.normal(0, 1, size=(out_channels, 1))

    # padding
    inputs_pad = np.pad(inputs, ((0,0), (padding, padding), (padding, padding)))

    ###########
    # 根据输入图像以及卷积核大小推测输出大小
    H_out = ... 
    W_out = ...
    outputs = np.zeros((F, H_out, W_out))
    ###########
    
    ###########
    # 模拟卷积核滑动完成卷积算子的实现
    
    ###########
    return outputs


def my_pool(inputs, pool_size, stride):
    C, H, W = inputs.shape

    ###########
    # 根据输入图像以及池化大小推测输出大小
    H_out = ...
    W_out = ...
    outputs = np.zeros((C, H_out, W_out)) 
    ###########

    ###########
    # 模拟池化核滑动完成池化算子的实现

    ###########
    return outputs