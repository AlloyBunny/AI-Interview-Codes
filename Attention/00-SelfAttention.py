import math

import torch
import torch.nn as nn

"""
SelfAttention
实现一个 SelfAttention 层，输入是 [batch_size, seq_len, embed_dim]，
输出是 [batch_size, seq_len, value_dim]。

q = q_proj(x)
k = k_proj(x)
v = v_proj(x)

scores = q @ k.T / sqrt(key_dim)
attn = softmax(scores)
output = attn @ v

注：
1. 这里实现的是 Encoder self-attention，没有 causal mask。
2. key_dim 和 value_dim 在数学上可以不同；MHA/GQA/MQA 里通常都等于 head_dim。
"""


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, key_dim, value_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # nn.Linear(in_dim, out_dim) 的权重矩阵维度是 [out_dim, in_dim]。
        self.q_proj = nn.Linear(embed_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)

    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "x.size(-1) 必须等于 embed_dim"

        q = self.q_proj(x)  # [batch_size, seq_len, key_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, key_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, value_dim]

        scores = q @ k.transpose(-1, -2) / math.sqrt(k.size(-1))  # [batch_size, seq_len, seq_len]
        attn = torch.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        output = attn @ v  # [batch_size, seq_len, value_dim]

        return output
