import torch
import torch.nn as nn

"""
Linear
线性层，也叫全连接层。

公式：
    y = x @ W.T + b

输入输出形状：
    x:      [..., in_features]
    weight: [out_features, in_features]
    bias:   [out_features]
    y:      [..., out_features]

注意：
nn.Linear(in_features, out_features)里的weight形状是[out_features, in_features]，
所以前向传播时要用weight.T，让输入x的最后一维in_features和weight.T的第一维对齐。
"""

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        """
        x: [..., in_features]
        """
        output = x @ self.weight.T # [..., in_features] @ [in_features, out_features] -> [..., out_features]

        if self.bias is not None:
            output = output + self.bias # [out_features]广播到[..., out_features]

        return output
