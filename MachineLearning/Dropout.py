import torch
import torch.nn as nn

"""
Dropout
训练时随机把一部分神经元置0，用来减少过拟合。

设drop概率为p，则保留概率为keep_prob = 1 - p。
这里实现的是inverted dropout：
    train: x * mask / keep_prob
    eval : x

为什么要除以keep_prob？
因为mask中只有keep_prob比例的位置为1，不缩放的话输出期望会变小。
除以keep_prob后，训练时输出的期望仍然等于原来的x，
所以推理时可以直接返回x，不需要再额外乘keep_prob。
"""

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1, "p 必须满足 0 <= p < 1"
        self.p = p
        self.keep_prob = 1 - p

    def forward(self, x):
        """
        x: 任意形状的tensor
        """
        if not self.training or self.p == 0: # self.training是nn.Module的属性，默认为True
            return x

        mask = torch.rand_like(x) < self.keep_prob # mask中的True表示保留该位置，torch.rand_like(x)是生成一个和x形状一样，每处的值为[0, 1)间的均匀随机数
        return x * mask / self.keep_prob # 除以(1-p)是为了保持期望不变

# 调用方式：
if __name__ == "__main__":
    x = torch.ones(2, 3)
    dropout = 0.1

    # 1. 直接指定p
    drop = Dropout(p=dropout) # 或者drop = Dropout(dropout)
    drop.train()
    y1 = drop(x)

    # 2. torch官方实现也是一样的
    torch_drop = nn.Dropout(dropout)
    y2 = torch_drop(x)

    print(y1)
    print(y2)