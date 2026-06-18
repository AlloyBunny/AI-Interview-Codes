import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        """
        x: [B, T, D], where:
            B: batch size
            T: sequence length
            D: hidden size
        """
        mean = x.mean(-1, keepdim = True)                # [B, T, D] -> [B, T, 1]
        var  = x.var(-1, keepdim = True, unbiased=False) # [B, T, D] -> [B, T, 1]

        x = (x - mean) * torch.rsqrt(var + self.eps) # rsqrt(x) = 1/sqrt(x)，但计算更快
        return x * self.weight + self.bias # 广播机制是从后往前广播，即[D]->[B, T, D]

"""
补充讲解：unbiased=False是啥？
本科的数理统计课学过，方差分两种：
1. population variance（总体方差）：
    分母是n，对应unbiased=False，用于做数值归一化
2. sample variance（样本方差）：
    分母是n-1，对应unbiased=True（pytorch默认值），用于做统计推断
"""