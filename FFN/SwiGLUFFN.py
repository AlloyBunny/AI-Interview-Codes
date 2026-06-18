import math

import torch
import torch.nn as nn

"""
SwiGLU FFN(架构图见 https://github.com/jingyaogong/minimind/blob/master/images/LLM-structure.jpg 的右边)
Transformer 里的前馈网络(Feed Forward Network)常见有两种写法：

1. 普通 FFN：
    FFN(x) = W2(activation(W1(x)))

2. SwiGLU FFN：
    SwiGLUFFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    SiLU(x) = x * sigmoid(x)

SwiGLU 的核心是“门控”：
    gate_proj(x) 生成门控分支
    up_proj(x)   生成内容分支
    两者逐元素相乘后，再用 down_proj 投影回 hidden_size

输入输出形状：
    x:      [..., hidden_size]
    gate:   [..., intermediate_size]
    up:     [..., intermediate_size]
    output: [..., hidden_size]

为什么默认 intermediate_size = 8 / 3 * hidden_size？
普通 FFN 常用 intermediate_size = 4 * hidden_size，参数量约为：
    hidden_size * 4h + 4h * hidden_size = 8h^2

SwiGLU 有 gate_proj、up_proj、down_proj 三个矩阵，参数量约为：
    hidden_size * d + hidden_size * d + d * hidden_size = 3hd

为了让参数量和普通 4h FFN 接近，令 3hd ≈ 8h^2，所以 d ≈ 8h/3。
LLaMA 系模型还会把 d 向上取整到 multiple_of 的倍数，方便硬件计算。
"""

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size=None,
        multiple_of=64,
        bias=False,
        dropout=0.1,
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = int(8 * hidden_size / 3) # 实践得到的经验最佳intermediate_size
            intermediate_size = int(math.ceil(intermediate_size/multiple_of) * multiple_of) # 向上取整到 multiple_of 的整数倍

        assert intermediate_size > 0, "intermediate_size 必须大于 0"

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [..., hidden_size]
        return: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        """
        gate = self.gate_proj(x)  # [..., intermediate_size]
        up = self.up_proj(x)      # [..., intermediate_size]

        hidden = self.act_fn(gate) * up
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output

# 调用方式：
if __name__ == "__main__":
    batch_size, seq_len, hidden_size = 2, 4, 128
    x = torch.randn(batch_size, seq_len, hidden_size)

    ffn = SwiGLUFFN(hidden_size=hidden_size, multiple_of=64, dropout=0.1)
    y = ffn(x)

    print(y.shape)  # torch.Size([2, 4, 128])，输出维度总是和输入维度一样