"""
00-Norm
减去均值，除以标准差
作用是：通过线性变换，使数据均值为0、方差为1，从而让不同特征具有一致的尺度，并让数据分布更适合神经网络优化。
"""
import torch
x = torch.tensor([1.,2.,3.,4.,5.])
mean = x.mean()
std = x.std()
eps = 1e-5 # eps是为了防止分母为0
x_norm = (x - mean) / (std + eps)
print(x_norm)

"""
输出：
tensor([-1.2649, -0.6325,  0.0000,  0.6325,  1.2649])
"""