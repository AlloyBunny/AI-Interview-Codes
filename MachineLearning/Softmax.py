import torch
import torch.nn as nn

"""
Softmax
输入是任意形状的tensor，沿着指定维度dim做归一化：

softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

为了数值稳定，实际计算时会先减去最大值：
softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

这样不会改变softmax的数值计算结果，因为分子分母都同时除以了exp(max(x))，
但是可以避免exp(x)在x很大时溢出。

注：两种方法算出来是完全一样的，这么做只是为了防溢出！
"""

def softmax(x, dim=-1):
    """
    函数版本的Softmax
    x: [..., D, ...]
    dim: 在哪一维上做softmax，常见的是最后一维dim=-1
    """
    x_max = x.max(dim=dim, keepdim=True).values # [..., D, ...] -> [..., 1, ...]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True) # 这里是逐元素除法，遇到广播

class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        x: [..., D, ...]
        dim: 在哪一维上做softmax，常见的是最后一维dim=-1
        """
        return softmax(x, dim=self.dim)

# 调用方式：
if __name__ == "__main__":
    x = torch.randn(2, 3)

    # 1. 调用这里手写的函数版本Softmax
    y1 = softmax(x, dim=-1)

    # 2. 调用这里手写的Module版本Softmax
    y2 = Softmax(dim=-1)(x)

    # 3. torch官方实现有两种常见调用方式
    y3 = torch.softmax(x, dim=-1)          # 函数式API
    y4 = nn.Softmax(dim=-1)(x)             # Module API

    print(y1)
    print(y2)
    print(y3)
    print(y4)

"""
四者调用方式不完全一样：
1. 手写函数版softmax(x, dim=-1) 和 torch.softmax(x, dim=-1) 一样，都是直接传x和dim
2. 手写Module版Softmax(dim=-1)(x) 和 nn.Softmax(dim=-1)(x) 一样，都是先创建一个module，再把x传进去

实际效果上，这四种写法都是沿着dim指定的维度做softmax。
在模型层里更常见的是nn.Softmax(dim=-1)这种Module写法；
在forward里临时算一下，更常见的是torch.softmax(x, dim=-1)这种函数式写法。
"""
