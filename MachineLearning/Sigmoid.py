import torch
import torch.nn as nn

"""
Sigmoid
输入是任意形状的tensor，对每个元素单独做映射：

sigmoid(x) = 1 / (1 + exp(-x))

Sigmoid会把任意实数压到(0, 1)之间：
    x很大时，sigmoid(x)接近1
    x很小时，sigmoid(x)接近0
    x等于0时，sigmoid(x)=0.5

所以它常用于二分类输出，把模型输出的logit转成概率。
"""

def sigmoid(x):
    """
    函数版本的Sigmoid
    x: 任意形状的tensor
    """
    return 1 / (1 + torch.exp(-x))

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: 任意形状的tensor
        """
        return sigmoid(x)

# 调用方式：
if __name__ == "__main__":
    x = torch.randn(2, 3)

    # 1. 调用这里手写的函数版本Sigmoid
    y1 = sigmoid(x)

    # 2. 调用这里手写的Module版本Sigmoid
    y2 = Sigmoid()(x)

    # 3. torch官方实现有两种常见调用方式
    y3 = torch.sigmoid(x)   # 函数式API
    y4 = nn.Sigmoid()(x)    # Module API

    print(y1)
    print(y2)
    print(y3)
    print(y4)

"""
Sigmoid需要写成nn.Module吗？
不一定。

如果只是临时算一下，用torch.sigmoid(x)最方便；
如果想把它当成模型中的一层，放进nn.Sequential或者写在__init__里，
就可以用nn.Sigmoid()这种Module写法。

比如：
    model = nn.Sequential(
        nn.Linear(10, 1),
        nn.Sigmoid()
    )

手写成class Sigmoid(nn.Module)的意义也是一样的：
它没有weight和bias，但它可以像一个普通网络层一样被调用。

和Softmax的区别：
1. Sigmoid是逐元素计算，每个位置互不影响
2. Softmax是在某个dim上一起归一化，这个dim上的结果加起来等于1

二分类常用Sigmoid，多分类常用Softmax。
"""
