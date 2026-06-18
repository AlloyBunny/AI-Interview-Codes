import torch
import torch.nn as nn

"""
FFN
Transformer 里的普通前馈网络(Feed Forward Network)，也常叫 MLP。

公式：
    FFN(x) = down_proj(activation(up_proj(x)))

常见写法：
    hidden_size -> intermediate_size -> hidden_size

输入输出形状：
    x:      [..., hidden_size]
    hidden: [..., intermediate_size]
    output: [..., hidden_size]

为什么默认 intermediate_size = 4 * hidden_size？
这是 Transformer 里最常见的 FFN 扩展比例：先把特征维度扩大到 4 倍，
经过非线性激活后，再投影回 hidden_size。

注：
原始 Transformer 常用 ReLU，BERT/GPT 等模型里更常见的是 GELU。
这里默认使用 GELU，面试时讲 ReLU/GELU 都可以，核心结构不变。
"""


class FFN(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size=None,
        bias=False,
        dropout=0.1,
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        assert intermediate_size > 0, "intermediate_size 必须大于 0"

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [..., hidden_size]
        return: down_proj(GELU(up_proj(x)))
        """
        hidden = self.up_proj(x)      # [..., intermediate_size]
        hidden = self.act_fn(hidden)  # [..., intermediate_size]
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output


# 调用方式：
if __name__ == "__main__":
    batch_size, seq_len, hidden_size = 2, 4, 128
    x = torch.randn(batch_size, seq_len, hidden_size)

    ffn = FFN(hidden_size=hidden_size, dropout=0.1)
    y = ffn(x)

    print(y.shape)  # torch.Size([2, 4, 128])，输出维度总是和输入维度一样
