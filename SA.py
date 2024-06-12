import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(self, Attention).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
    
    def forward(self, q, k, v, mask=None):
        B, C, N = q.shape()

        # 拆分QKV为多头
        q = q.view(B, self.num_heads, self.head_dim, N) # [B num_heads head_dim N]
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)

        atten = torch.matmul(q.transpose(-2,-1), k) / torch.sqrt(self.head_dim) # [B num_heads N N]

        atten_weight = F.softmax(atten, dim=-1) # [B num_heads N N]
        output = torch.matmul(atten_weight, v) # [B num_heads self.head_dim N]

        output = output.contiguous().view(B, -1, N)

        return output