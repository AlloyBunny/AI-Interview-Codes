import torch
import torch.nn as nn

"""
01-BatchNorm
实现一个BatchNorm层，输入是(B, C, H, W)，输出是(B, C, H, W)，其中B是Batch Size，C是Channel，H是Height，W是Width。
其中，BatchNorm的公式为：
y = (x - mean) / sqrt(var + 1e-5) * gamma + beta
其中，mean是x的均值，var是x的方差，gamma和beta是可学习的参数，1e-5是为了防止分母为0。
"""
class BatchNorm(nn.Module):
    def __init__(self, C): 
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))
    
    def forward(self, x):
        mean = x.mean((0,2,3), keepdim=True)
        var  = x.var((0,2,3), keepdim=True, unbiased=False) # unbiased=False是总体方差，而不是无偏估计（因为BN不是做统计推断，而是需要一个稳定的尺度估计）

        x = (x-mean) / torch.sqrt(var+1e-5) # 注意这里，x是(B, C, H, W)，mean和var是(1, C, 1, 1)，torch的广播机制会让实际计算的时候mean和var变为(B, C, H, W)

        return self.gamma[None,:,None,None]*x + self.beta[None,:,None,None] # self.gamma[None,:,None,None]是手动把(C)变为(1, C, 1, 1)，乘上x的时候，也有广播机制